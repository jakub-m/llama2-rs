use clap::Parser;
use memmap2::{Mmap, MmapOptions};
use std::{
    fs::{File, metadata},
    io::{BufReader, Read},
};

struct TokenIndex<'a> {
    token_str: &'a str,
    id: usize,
}

struct Tokenizer {
    // Note: use ascii string (char) instead of unicode str.
    vocab: Vec<String>,
    vocab_scores: Vec<f32>,
    //sorted_vocab: Vec<TokenIndex<'a>>,
    vocab_size: usize,
    max_token_length: usize,
    //Note: use memory-continuous representation of those byte pieces, with simpler strings.
    byte_pieces: [Box<String>; 256],
}

impl Tokenizer {
    fn build(path: &str, vocab_size: usize) -> Tokenizer {
        let byte_pieces: [Box<String>; 256] =
            std::array::from_fn(|i| Box::new((i as u8 as char).to_string()));
        //dbg!(&byte_pieces);
        let mut vocab = vec!["".to_string(); vocab_size];
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
            vocab[i] = s;
        }

        Tokenizer {
            vocab_size,
            vocab,
            vocab_scores,
            byte_pieces,
            max_token_length,
        }
    }
}

struct Transformer<'a> {
    /// the hyperparameters of the architecture (the blueprint)
    config: Config,
    /// the weights of the model
    weights: TransformerWeights<'a>,
    file: File,
    //    RunState state; // buffers for the "wave" of activations in the forward pass
    //    // some more state needed to properly clean up the memory mapping (sigh)
    //    int fd; // file descriptor for memory mapping
    //    float* data; // memory mapped data pointer
    //    ssize_t file_size; // size of the checkpoint file in bytes
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
        // TODO HERE malloc
        //todo!("continue malloc...")
        //malloc_run_state(&t->state, &t->config);
        let (config, checkpoint) = Self::read_checkpoint(checkpoint_path);
        Transformer {
            config,
            weights: checkpoint.tw,
            file: checkpoint.file,
        }
    }

    fn read_checkpoint<'c>(checkpoint_path: &str) -> (Config, Checkpoint<'c>) {
        let mut config: Config;
        let shared_weights: bool;
        {
            let file = File::open(checkpoint_path).unwrap();
            let mut file = BufReader::new(file);
            // read in the config header
            let mut buf = [0u8; size_of::<Config>()];
            file.read_exact(&mut buf).unwrap();
            unsafe {
                config = std::mem::transmute(buf);
            }
            eprintln!("{config:?}");
            shared_weights = config.vocab_size > 0;
            config.vocab_size = config.vocab_size.abs();
        }

        // memory map the Transformer weights into the data pointer
        let file = File::open(checkpoint_path).unwrap();
        let data = unsafe {
            MmapOptions::new().map_copy_read_only(&file).unwrap() // open in read only mode, equivalent of mmap PROT_READ, MAP_PRIVATE
        };

        let data: &[f32] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / size_of::<f32>())
        };

        let data = &data[size_of::<Config>() / size_of::<f32>()..];
        let transformer_weights = memory_map_weights(&config, data, shared_weights);
        (
            config,
            Checkpoint {
                file,
                tw: transformer_weights,
            },
        )
    }
}

/// A helper to return mmap-ed memory and the mmap-ed file.
struct Checkpoint<'a> {
    file: File,
    tw: TransformerWeights<'a>,
}

fn memory_map_weights<'a>(
    p: &Config,
    ptr: &'a [f32],
    shared_weights: bool,
) -> TransformerWeights<'a> {
    let head_size = p.dim / p.n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    let token_embedding_table = ptr; // (vocab_size, dim)
    let ptr: &[f32] = &ptr[(p.vocab_size * p.dim) as usize..];

    // weights for rmsnorms
    let rms_att_weight = ptr; // (layer, dim) rmsnorm weights
    let ptr: &[f32] = &ptr[(p.n_layers * p.dim) as usize..];

    // weights for matmuls. note dim == n_heads * head_size
    let wq = ptr; // (layer, dim, n_heads * head_size)
    let ptr: &[f32] = &ptr[(p.n_layers * p.dim * (p.n_heads * head_size)) as usize..];

    let wk = ptr; // (layer, dim, n_kv_heads * head_size)
    let ptr: &[f32] = &ptr[(p.n_layers * p.dim * (p.n_kv_heads * head_size)) as usize..];

    let wv = ptr; //  (layer, dim, n_kv_heads * head_size)
    let ptr: &[f32] = &ptr[(p.n_layers * p.dim * (p.n_kv_heads * head_size)) as usize..];

    let wo = ptr; // (layer, n_heads * head_size, dim)
    let ptr: &[f32] = &ptr[(p.n_layers * (p.n_heads * head_size) * p.dim) as usize..];

    let rms_ffn_weight = ptr;
    let ptr: &[f32] = &ptr[(p.n_layers * p.dim) as usize..];

    // weights for ffn
    let w1 = ptr; // (layer, hidden_dim, dim)
    let ptr: &[f32] = &ptr[(p.n_layers * p.dim * p.hidden_dim) as usize..];

    let w2 = ptr; // (layer, dim, hidden_dim);
    let ptr: &[f32] = &ptr[(p.n_layers * p.hidden_dim * p.dim) as usize..];

    let w3 = ptr; // (layer, hidden_dim, dim)
    let ptr: &[f32] = &ptr[(p.n_layers * p.dim * p.hidden_dim) as usize..];

    // final rmsnorm
    let rms_final_weight = ptr; // (dim,)
    let ptr: &[f32] = &ptr[(p.dim) as usize..];
    let ptr: &[f32] = &ptr[(p.seq_len * head_size / 2) as usize..]; // skip what used to be freq_cis_real (for RoPE)
    let ptr: &[f32] = &ptr[(p.seq_len * head_size / 2) as usize..]; // skip what used to be freq_cis_imag (for RoPE)

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

#[repr(C)]
#[derive(Debug)]
struct Config {
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

fn main() {
    let args = Args::parse();
    println!("Hello, world! {args:?}");

    Transformer::build(&args.checkpoint_path);
    // TODO missing stuff from run.c
    Tokenizer::build(&args.tokenizer_path, 32000);
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(short = 'z')]
    tokenizer_path: String,
    checkpoint_path: String,
}
