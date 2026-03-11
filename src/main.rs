use std::collections::VecDeque;
use std::f32::EPSILON;
use std::io::BufRead;
use std::path::Path;

use ndarray::{Array1, Array2, Axis};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tokenizers::utils::padding::PaddingStrategy;
use tokenizers::utils::truncation::TruncationStrategy;
use tokenizers::Tokenizer;

use argh::FromArgs;

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
    /// get model information
    #[argh(switch)]
    info: bool,
    /// threshold
    #[argh(option, default = "default_threshold()")]
    threshold: f32,
    /// reset context on paragraph (empty line)
    #[argh(switch)]
    paragraph: bool,
}

fn default_context() -> usize {
    1
}

fn default_threshold() -> f32 {
    -1.0
}

fn main() -> anyhow::Result<()> {
    // Get CLI args
    let args: CliArgs = argh::from_env();
    let model_path = Path::new(&args.model_path);

    // Set ORT backend API & load model
    ort::set_api(ort_tract::api());
    let mut model = Model::new(model_path)?;

    if args.info {
        model.show_info();
        return Ok(());
    }

    // Get embedding for search concept
    let search_v = model.run(args.concept.as_str())?;

    let mut line_buf = VecDeque::<String>::new();

    // Read from stdin
    let stdin = std::io::stdin().lock();
    let mut lines = stdin.lines();
    let mut line_num = 1_usize;

    while let Some(Ok(line)) = lines.next() {
        if line.is_empty() && args.paragraph {
            line_buf.clear();
        }

        // Push line into context buffer
        line_buf.push_back(line.clone());
        if line_buf.len() > args.context {
            line_buf.pop_front();
        }

        // Run model on context
        let context = line_buf.make_contiguous().join(" ");
        let context_v = model.run(&context)?;

        // Check similarity (arrays are already normalised from .run())
        let similarity = cosine_similarity(&search_v, &context_v, false);

        if similarity > args.threshold {
            println!("[{:-4}/{:5.3}] {}", line_num, similarity, line);
        }

        // let p = ndarray::stack(Axis(0), &[search_v.view(), merged_v.view()])?;
        // print_array2(&pairwise_similarity(&p, true));

        line_num += 1;
    }
    Ok(())
}

pub struct Model {
    session: Session,
    tokenizer: Tokenizer,
    embedding_len: usize,
}

impl Model {
    /// Create new model
    pub fn new(path: &Path) -> anyhow::Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {}", e))?
            .commit_from_file(path.join("model.onnx"))
            .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;

        // Check model has a sentence_embedding output and extract length
        let embedding_len = match session
            .outputs()
            .iter()
            .find(|o| o.name() == "sentence_embedding")
        {
            Some(o) => match o.dtype() {
                ort::value::ValueType::Tensor { ty, shape, .. } => {
                    match ty {
                        ort::value::TensorElementType::Float32 => {}
                        _ => anyhow::bail!("Model Error: Output must be f32"),
                    }
                    let dim = shape[1];
                    if dim > 0 {
                        dim as usize
                    } else {
                        anyhow::bail!("Model Error: Invalid output dimension: {dim}")
                    }
                }
                _ => anyhow::bail!("Model Error: Invalid datatype"),
            },
            None => anyhow::bail!("Model Error: No sentence_embedding output"),
        };

        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("Tokenizer: {e}"))?;

        // We expect tokenizer to take care of overflow + padding
        //
        // <<tokenizer.json>>
        // ...
        //    "truncation": {
        //      "direction": "Right",
        //      "max_length": 128,
        //      "strategy": "LongestFirst",
        //      "stride": 0
        //    },
        //    "padding": {
        //      "strategy": {
        //        "Fixed": 128
        //      },
        //      "direction": "Right",
        //      "pad_to_multiple_of": null,
        //      "pad_id": 0,
        //      "pad_type_id": 0,
        //      "pad_token": "[PAD]"
        //  },
        //  ...

        // Check tokenizer settings
        let truncation = tokenizer
            .get_truncation()
            .ok_or_else(|| anyhow::anyhow!("get_truncation"))?;
        match truncation.strategy {
            TruncationStrategy::LongestFirst | TruncationStrategy::OnlyFirst => {}
            _ => anyhow::bail!("Invalid Truncation Strategy"),
        };
        let padding = tokenizer
            .get_padding()
            .ok_or_else(|| anyhow::anyhow!("get_padding"))?;
        let padding_len = match padding.strategy {
            PaddingStrategy::Fixed(n) => n,
            _ => anyhow::bail!("Invalid Padding Strategy"),
        };
        if padding_len != truncation.max_length {
            anyhow::bail!("Padding/Truncation length dont match");
        }

        Ok(Self {
            session,
            tokenizer,
            embedding_len,
        })
    }

    /// Show model info
    pub fn show_info(&self) {
        for inlet in self.session.inputs() {
            println!("Input: name={}, dtype={:?}", inlet.name(), inlet.dtype(),);
        }
        for outlet in self.session.outputs() {
            println!("Output: name={}, dtype={:?}", outlet.name(), outlet.dtype(),);
        }
    }

    /// Run model against input string and return merged/normalised embedding
    ///
    /// (If the context size exceeds the tokeniser max_length we extract
    /// the overflow into multiple encodings, run the model against all of
    /// these and then take the mean of the output vector)
    pub fn run(&mut self, input: &str) -> anyhow::Result<Array1<f32>> {
        if input.is_empty() {
            return Ok(Array1::zeros(self.embedding_len));
        }

        let mut encodings = self
            .tokenizer
            .encode(input, true)
            .map_err(|e| anyhow::anyhow!("Encode: {e}"))?;

        // Handle overflow - extract main + overflow encodings
        let overflow = encodings.take_overflowing();
        let mut encodings_v = vec![encodings];
        encodings_v.extend(overflow);

        let batch_size = encodings_v.len();
        let seq_length = encodings_v[0].len();

        let input_ids = Tensor::from_array((
            [batch_size, seq_length],
            encodings_v
                .iter()
                .flat_map(|e| e.get_ids().iter().map(|&x| x as i64))
                .collect::<Box<[i64]>>(),
        ))?;
        let attention_mask = Tensor::from_array((
            [batch_size, seq_length],
            encodings_v
                .iter()
                .flat_map(|e| e.get_attention_mask().iter().map(|&x| x as i64))
                .collect::<Box<[i64]>>(),
        ))?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids,
            "attention_mask" => attention_mask
        ])?;

        // 2D array output
        let embeddings = outputs["sentence_embedding"]
            .try_extract_array::<f32>()?
            .into_dimensionality::<ndarray::Ix2>()?;

        // Collapse embeddings to Array1 (mean of chunks)
        let merged = match embeddings.mean_axis(Axis(0)) {
            Some(v) => v,
            None => Array1::zeros(self.embedding_len), // Empty embedding
        };

        // Return normalised embedding
        let norm = merged.dot(&merged).sqrt();
        if norm < EPSILON {
            Ok(Array1::zeros(self.embedding_len))
        } else {
            Ok(merged / norm)
        }
    }
}

fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>, normalise: bool) -> f32 {
    let dot = a.dot(b);
    if normalise {
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();

        if norm_a <= EPSILON || norm_b <= EPSILON {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    } else {
        dot
    }
}

#[allow(unused)]
fn pairwise_similarity(embeddings: &Array2<f32>, normalise: bool) -> Array2<f32> {
    let n = embeddings.nrows();
    let mut sim = Array2::zeros((n, n));

    let normalised = if normalise {
        // XXX Fix EPSILON handling XXX
        // Normalise rows to unit length
        //   - Calculate norm_l2 by row -> Array1
        //   - Rotate to Array2 and multiply embeddings (broadcast division)
        let norm = embeddings.map_axis(Axis(1), |row| row.dot(&row).sqrt());
        let norm = norm.to_shape((norm.len(), 1)).unwrap(); // Safe as we know sizes
        &(embeddings / (norm + EPSILON)).to_owned() // Add EPSILON to avoid Div0
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
