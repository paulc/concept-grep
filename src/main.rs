use std::collections::VecDeque;
use std::io::BufRead;
use std::path::Path;

use ndarray::{Array1, Array2, Axis};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use argh::FromArgs;

const CHUNK_SIZE: usize = 200; // Max 256 tokens
const EPSILON: f32 = 1e-9; // Epsilon value to avoid Div0

#[derive(FromArgs)]
/// CLI Arguments
struct CliArgs {
    /// concept
    #[argh(option, short = 'c')]
    concept: String,
    /// model path
    #[argh(option, default = "String::from(\"models/MiniLM-L6-v2\")")]
    model_path: String,
    /// context lines
    #[argh(option, default = "default_context()")]
    context: usize,
}

fn default_context() -> usize {
    1
}

fn main() -> anyhow::Result<()> {
    // Get CLI args
    let args: CliArgs = argh::from_env();
    let model_path = Path::new(&args.model_path);

    // Set ORT backend API & load model
    ort::set_api(ort_tract::api());
    let mut model = Model::new(model_path)?;

    // Get enbedding for search concept
    let search_v = model.run(&[args.concept])?.row(0).to_owned();

    let mut line_buf = VecDeque::<String>::new();

    // Read from stdin
    let stdin = std::io::stdin().lock();
    let mut lines = stdin.lines();
    while let Some(Ok(line)) = lines.next() {
        // Push line into context buffer
        line_buf.push_back(line.clone());
        if line_buf.len() > args.context {
            line_buf.pop_front();
        }
        // Run model on chunked context imput
        let context_v = model.run(&chunk_input(line_buf.make_contiguous()))?;
        // Collapse embeddings to Array1 (mean of chunks)
        let merged_v = context_v
            .mean_axis(Axis(0))
            .ok_or_else(|| anyhow::anyhow!("mean_axis"))?;
        // Check similarity (need to normalise as merging will impact magnitide)
        let similarity = cosine_similarity(&search_v, &merged_v, true);
        println!("[{:5.3}] {}", similarity, line);

        // let p = ndarray::stack(Axis(0), &[search_v.view(), merged_v.view()])?;
        // print_array2(&pairwise_similarity(&p, true));
    }
    Ok(())
}

pub struct Model {
    session: Session,
    tokenizer: Tokenizer,
}

impl Model {
    pub fn new(path: &Path) -> anyhow::Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {}", e))?
            .commit_from_file(path.join("model.onnx"))
            .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;
        // Assumes that tokenizer.json specifies truncation/padding
        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("Tokenizer: {e}"))?;
        Ok(Self { session, tokenizer })
    }
    pub fn show_info(&self) {
        for outlet in self.session.inputs().iter() {
            println!("Input: {:?}", outlet);
        }
        for outlet in self.session.outputs().iter() {
            println!("Output: {:?}", outlet);
        }
    }
    pub fn run(&mut self, inputs: &[String]) -> anyhow::Result<Array2<f32>> {
        if inputs.is_empty() {
            return Ok(Array2::zeros((0, 384)));
        }
        let encodings = self
            .tokenizer
            .encode_batch(inputs.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Encode Batch: {e}"))?;
        let batch_size = encodings.len();
        let seq_length = encodings[0].len(); // 128

        let input_ids = Tensor::from_array((
            [batch_size, seq_length],
            encodings
                .iter()
                .flat_map(|e| e.get_ids().iter().map(|&x| x as i64))
                .collect::<Box<[i64]>>(),
        ))?;
        let attention_mask = Tensor::from_array((
            [batch_size, seq_length],
            encodings
                .iter()
                .flat_map(|e| e.get_attention_mask().iter().map(|&x| x as i64))
                .collect::<Box<[i64]>>(),
        ))?;

        let outputs = self.session.run(ort::inputs![input_ids, attention_mask])?;

        // Force 2D array output
        let embeddings = outputs["sentence_embedding"]
            .try_extract_array::<f32>()?
            .into_dimensionality::<ndarray::Ix2>()?
            .to_owned();

        Ok(embeddings)
    }
}

fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>, normalise: bool) -> f32 {
    let dot = a.dot(b);
    if normalise {
        let norm_a = a.mapv(|x| x * x).sum().sqrt();
        let norm_b = b.mapv(|x| x * x).sum().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    } else {
        dot
    }
}

fn chunk_input(lines: &[String]) -> Vec<String> {
    lines
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .chunks(CHUNK_SIZE)
        .map(|c| c.join(" "))
        .collect()
}

#[allow(unused)]
fn pairwise_similarity(embeddings: &Array2<f32>, normalise: bool) -> Array2<f32> {
    let n = embeddings.nrows();
    let mut sim = Array2::zeros((n, n));

    let normalised = if normalise {
        // Normalise rows to unit length
        //   - Calculate norm_l2 by row -> Array1
        //   - Rotate to Array2 and multiply embeddings (broadcast division)
        let norm = embeddings.map_axis(Axis(1), |row| row.mapv(|x| x * x).sum().sqrt());
        let norm = norm.to_shape((norm.len(), 1)).unwrap(); // Safe as we know sizes
        &(embeddings / norm + EPSILON).to_owned() // Ass EPSILON to avoid Div0
    } else {
        embeddings
    };
    // Compute matrix with dot products
    for i in 0..n {
        for j in i..n {
            let val = normalised.row(i).dot(&normalised.row(j));
            sim[[i, j]] = val;
            sim[[j, i]] = val; // Symmetry
        }
    }
    sim
}

#[allow(unused)]
fn print_array2(arr: &Array2<f32>) {
    for row in arr.outer_iter() {
        for val in row {
            print!("{:5.2} ", val);
        }
        println!();
    }
}
