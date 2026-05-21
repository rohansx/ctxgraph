//! Piece 3 — Relation-vocabulary embeddings.
//!
//! At construction: embed each of the 10 universal-schema relation names + their
//! descriptions, cache the resulting 384-dim vectors.
//!
//! At query time: embed the user's verb, cosine-match against the cached
//! vectors, return the best relation name + its score.
//!
//! Source: `docs/CLARITY.md` § 3 / Piece 3. Validated by the Python prototype
//! in `scripts/proto_relation_match.py` (83% accuracy on 24 verb variations).
//!
//! ```ignore
//! use ctxgraph_extract::relation_match::RelationMatcher;
//! use ctxgraph_extract::schema::ExtractionSchema;
//!
//! let schema = ExtractionSchema::from_universal_toml(include_str!(
//!     "../schemas/universal.toml"
//! ))?;
//! let matcher = RelationMatcher::build_from_schema(&schema)?;
//! let m = matcher.resolve("relies on")?;
//! assert_eq!(m.relation, "depends_on");
//! ```

use std::path::PathBuf;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use crate::schema::ExtractionSchema;

const EMBEDDING_DIM: usize = 384;

#[derive(Debug, thiserror::Error)]
pub enum RelationMatchError {
    #[error("failed to initialize embedding model: {0}")]
    ModelInit(String),
    #[error("failed to encode text: {0}")]
    Encoding(String),
    #[error("no relations loaded — call build_from_schema first")]
    NotInitialized,
    #[error("empty input text")]
    EmptyInput,
}

#[derive(Debug, Clone)]
pub struct RelationMatch {
    pub relation: String,
    pub score: f32,
    pub runner_up: String,
    pub runner_up_score: f32,
}

pub struct RelationMatcher {
    model: TextEmbedding,
    relation_names: Vec<String>,
    relation_vectors: Vec<Vec<f32>>, // shape: (N, 384)
}

impl RelationMatcher {
    /// Build a matcher from any `ExtractionSchema`. Embeds each relation name +
    /// description and caches the vectors.
    pub fn build_from_schema(schema: &ExtractionSchema) -> Result<Self, RelationMatchError> {
        let model = TextEmbedding::try_new(InitOptions::new(EmbeddingModel::AllMiniLML6V2))
            .map_err(|e| RelationMatchError::ModelInit(e.to_string()))?;
        Self::build_inner(model, schema)
    }

    /// Build a matcher with a custom fastembed cache directory.
    pub fn build_from_schema_with_cache(
        schema: &ExtractionSchema,
        cache_dir: PathBuf,
    ) -> Result<Self, RelationMatchError> {
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_cache_dir(cache_dir),
        )
        .map_err(|e| RelationMatchError::ModelInit(e.to_string()))?;
        Self::build_inner(model, schema)
    }

    fn build_inner(
        model: TextEmbedding,
        schema: &ExtractionSchema,
    ) -> Result<Self, RelationMatchError> {
        let mut names: Vec<String> = Vec::new();
        let mut texts: Vec<String> = Vec::new();
        for (name, spec) in &schema.relation_types {
            names.push(name.clone());
            texts.push(format!("{}: {}", name, spec.description));
        }
        if names.is_empty() {
            return Err(RelationMatchError::NotInitialized);
        }
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let vectors = model
            .embed(text_refs, None)
            .map_err(|e| RelationMatchError::Encoding(e.to_string()))?;
        // Sanity-check dimensions
        for v in &vectors {
            debug_assert_eq!(v.len(), EMBEDDING_DIM);
        }
        Ok(Self {
            model,
            relation_names: names,
            relation_vectors: vectors,
        })
    }

    /// Number of relation types this matcher knows about.
    pub fn len(&self) -> usize {
        self.relation_names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.relation_names.is_empty()
    }

    /// Resolve a natural-language verb to the closest typed relation.
    /// Returns the top-1 plus the runner-up for debugging / threshold logic.
    pub fn resolve(&self, user_verb: &str) -> Result<RelationMatch, RelationMatchError> {
        let verb = user_verb.trim();
        if verb.is_empty() {
            return Err(RelationMatchError::EmptyInput);
        }
        let query = self
            .model
            .embed(vec![verb], None)
            .map_err(|e| RelationMatchError::Encoding(e.to_string()))?
            .pop()
            .ok_or_else(|| RelationMatchError::Encoding("empty embedding result".into()))?;

        let mut scored: Vec<(usize, f32)> = self
            .relation_vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_similarity(&query, v)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // After sort, scored[0] is best, scored[1] is runner-up
        let (best_idx, best_score) = scored[0];
        let (runner_idx, runner_score) = scored.get(1).copied().unwrap_or((best_idx, best_score));

        Ok(RelationMatch {
            relation: self.relation_names[best_idx].clone(),
            score: best_score,
            runner_up: self.relation_names[runner_idx].clone(),
            runner_up_score: runner_score,
        })
    }
}

/// Compute cosine similarity between two f32 vectors. Returns 0.0 if either
/// vector has zero magnitude or the lengths differ.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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
