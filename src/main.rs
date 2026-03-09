use std::path::Path;

use ndarray::{Array1, Array2, Axis};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tokenizers::Tokenizer;

fn main() -> anyhow::Result<()> {
    // Set ORT backend API
    ort::set_api(ort_tract::api());

    let mut model = Model::new("models/MiniLM-L6-v2/model.onnx")?;

    let embeddings = model.encode_batch(&[
        "Hello there, I'm a llama!",
        "Pass the aardvark, vicar",
        "What is the speed of a flying capybara?",
        "Llamas are very nice animals",
        "My favourite animal is a llama",
        "Cute animals",
    ])?;

    let sim = pairwise_similarity(&embeddings);
    print_array2(&sim);

    Ok(())
}

pub struct Model {
    session: Session,
    tokenizer: Tokenizer,
}

impl Model {
    pub fn new(path: &str) -> anyhow::Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {}", e))?
            .commit_from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;
        let tokenizer = Tokenizer::from_file(Path::new("models/MiniLM-L6-v2/tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("Tokenizer: {e}"))?;
        Ok(Self { session, tokenizer })
    }
    pub fn outputs(&self) {
        for outlet in self.session.outputs().iter() {
            println!("Outlet: {:?}", outlet);
        }
    }
    pub fn encode_batch(&mut self, inputs: &[&str]) -> anyhow::Result<Array2<f32>> {
        let encodings = self
            .tokenizer
            .encode_batch(inputs.to_vec(), false)
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

fn _cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot = a.dot(b);
    let norm_a = a.mapv(|x| x * x).sum().sqrt();
    let norm_b = b.mapv(|x| x * x).sum().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn pairwise_similarity(embeddings: &Array2<f32>) -> Array2<f32> {
    let n = embeddings.nrows();
    let mut sim = Array2::zeros((n, n));

    // 1. Pre-normalize rows to unit length (L2)
    let mut normalized = embeddings.clone();
    for mut row in normalized.axis_iter_mut(Axis(0)) {
        let norm = row.mapv(|x| x * x).sum().sqrt();
        if norm > 0.0 {
            row.mapv_inplace(|x| x / norm);
        }
    }
    // 2. Compute dot products (symmetric matrix)
    for i in 0..n {
        for j in i..n {
            let val = normalized.row(i).dot(&normalized.row(j));
            sim[[i, j]] = val;
            sim[[j, i]] = val; // Symmetry
        }
    }
    sim
}

fn print_array2(arr: &Array2<f32>) {
    for row in arr.axis_iter(Axis(0)) {
        for val in row {
            print!("{:5.2} ", val);
        }
        println!();
    }
}
