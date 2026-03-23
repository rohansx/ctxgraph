use std::path::PathBuf;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use crate::EmbedError;

pub struct EmbedEngine {
    model: TextEmbedding,
}

impl EmbedEngine {
    /// Initialize with default cache directory (~/.cache/fastembed).
    pub fn new() -> Result<Self, EmbedError> {
        let model = TextEmbedding::try_new(InitOptions::new(EmbeddingModel::AllMiniLML6V2))
            .map_err(|e| EmbedError::ModelInit(e.to_string()))?;
        Ok(Self { model })
    }

    /// Initialize with a custom model cache directory.
    pub fn new_with_cache(cache_dir: PathBuf) -> Result<Self, EmbedError> {
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_cache_dir(cache_dir),
        )
        .map_err(|e| EmbedError::ModelInit(e.to_string()))?;
        Ok(Self { model })
    }

    /// Embed a single text string. Returns a 384-dimensional vector.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let mut batch = self
            .model
            .embed(vec![text], None)
            .map_err(|e| EmbedError::Encoding(e.to_string()))?;
        batch
            .pop()
            .ok_or_else(|| EmbedError::Encoding("empty embedding result".to_string()))
    }

    /// Embed a batch of texts. Returns one vector per input text.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.model
            .embed(texts.to_vec(), None)
            .map_err(|e| EmbedError::Encoding(e.to_string()))
    }

    /// Compute cosine similarity between two f32 vectors.
    /// Returns 0.0 if either vector has zero magnitude.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag_a == 0.0 || mag_b == 0.0 {
            0.0
        } else {
            dot / (mag_a * mag_b)
        }
    }
}
