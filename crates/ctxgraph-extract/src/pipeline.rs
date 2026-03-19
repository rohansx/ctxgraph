use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};

use crate::ner::{ExtractedEntity, NerEngine, NerError};
use crate::rel::{ExtractedRelation, RelEngine, RelError};
use crate::schema::{ExtractionSchema, SchemaError};
use crate::temporal::{self, TemporalResult};

/// Complete result of running the extraction pipeline on a piece of text.
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    pub entities: Vec<ExtractedEntity>,
    pub relations: Vec<ExtractedRelation>,
    pub temporal: Vec<TemporalResult>,
}

/// The extraction pipeline orchestrates NER, relation extraction, and temporal parsing.
///
/// Created once and reused across multiple episodes. Model loading happens at construction
/// time (~100-500ms), but subsequent inference calls are fast (<15ms).
pub struct ExtractionPipeline {
    schema: ExtractionSchema,
    ner: NerEngine,
    rel: RelEngine,
    confidence_threshold: f64,
}

impl ExtractionPipeline {
    /// Create a new extraction pipeline.
    ///
    /// - `schema`: Entity/relation type definitions.
    /// - `models_dir`: Directory containing ONNX model files.
    /// - `confidence_threshold`: Minimum confidence to keep an extraction (default: 0.5).
    pub fn new(
        schema: ExtractionSchema,
        models_dir: &Path,
        confidence_threshold: f64,
    ) -> Result<Self, PipelineError> {
        // Locate NER model files (span-based GLiNER v2.1)
        let ner_model = find_ner_model(models_dir)?;
        let ner_tokenizer = find_tokenizer(models_dir, "gliner")?;

        let ner = NerEngine::new(&ner_model, &ner_tokenizer).map_err(PipelineError::Ner)?;

        // Locate relation model files (multitask GLiNER) — optional
        let rel_model = find_rel_model(models_dir);
        let rel_tokenizer = find_tokenizer(models_dir, "multitask").ok();

        let rel = RelEngine::new(
            rel_model.as_deref(),
            rel_tokenizer.as_deref(),
        )
        .map_err(PipelineError::Rel)?;

        Ok(Self {
            schema,
            ner,
            rel,
            confidence_threshold,
        })
    }

    /// Create a pipeline with default settings.
    ///
    /// Uses `ExtractionSchema::default()` and 0.5 confidence threshold.
    pub fn with_defaults(models_dir: &Path) -> Result<Self, PipelineError> {
        Self::new(ExtractionSchema::default(), models_dir, 0.5)
    }

    /// Extract entities, relations, and temporal expressions from text.
    pub fn extract(
        &self,
        text: &str,
        reference_time: DateTime<Utc>,
    ) -> Result<ExtractionResult, PipelineError> {
        // Step 1: NER — extract entities
        let labels: Vec<&str> = self.schema.entity_labels();
        let mut entities = self
            .ner
            .extract(text, &labels)
            .map_err(PipelineError::Ner)?;

        // Filter by confidence
        entities.retain(|e| e.confidence >= self.confidence_threshold);

        // Step 2: Relation extraction
        let mut relations = self
            .rel
            .extract(text, &entities, &self.schema)
            .map_err(PipelineError::Rel)?;

        // Filter by confidence
        relations.retain(|r| r.confidence >= self.confidence_threshold);

        // Step 3: Temporal parsing
        let temporal = temporal::parse_temporal(text, reference_time);

        Ok(ExtractionResult {
            entities,
            relations,
            temporal,
        })
    }

    /// Get the schema used by this pipeline.
    pub fn schema(&self) -> &ExtractionSchema {
        &self.schema
    }

    /// Get the confidence threshold.
    pub fn confidence_threshold(&self) -> f64 {
        self.confidence_threshold
    }
}

/// Find the NER ONNX model file in the models directory.
///
/// Looks for these files in order:
/// 1. `gliner_large-v2.1/onnx/model_int8.onnx` (quantized, recommended)
/// 2. `gliner_large-v2.1/onnx/model.onnx` (full precision)
/// 3. `gliner2-large-q8.onnx` (legacy flat layout)
fn find_ner_model(models_dir: &Path) -> Result<PathBuf, PipelineError> {
    let candidates = [
        models_dir.join("gliner_large-v2.1/onnx/model_int8.onnx"),
        models_dir.join("gliner_large-v2.1/onnx/model.onnx"),
        models_dir.join("gliner2-large-q8.onnx"),
    ];

    for c in &candidates {
        if c.exists() {
            return Ok(c.clone());
        }
    }

    Err(PipelineError::ModelNotFound {
        model: "GLiNER v2.1 NER".into(),
        searched: candidates.iter().map(|p| p.display().to_string()).collect(),
    })
}

/// Find the relation extraction model (multitask GLiNER).
fn find_rel_model(models_dir: &Path) -> Option<PathBuf> {
    let candidates = [
        models_dir.join("gliner-multitask-large-v0.5/onnx/model.onnx"),
        models_dir.join("gliner-multitask-large.onnx"),
        models_dir.join("glirel-large.onnx"),
    ];

    candidates.into_iter().find(|c| c.exists())
}

/// Find a tokenizer.json file associated with a model.
fn find_tokenizer(models_dir: &Path, prefix: &str) -> Result<PathBuf, PipelineError> {
    let candidates = if prefix == "gliner" {
        vec![
            models_dir.join("gliner_large-v2.1/tokenizer.json"),
            models_dir.join("tokenizer.json"),
        ]
    } else {
        vec![
            models_dir.join("gliner-multitask-large-v0.5/tokenizer.json"),
            models_dir.join("tokenizer.json"),
        ]
    };

    for c in &candidates {
        if c.exists() {
            return Ok(c.clone());
        }
    }

    Err(PipelineError::ModelNotFound {
        model: format!("{prefix} tokenizer").into(),
        searched: candidates.iter().map(|p| p.display().to_string()).collect(),
    })
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("NER error: {0}")]
    Ner(#[from] NerError),

    #[error("relation extraction error: {0}")]
    Rel(#[from] RelError),

    #[error("schema error: {0}")]
    Schema(#[from] SchemaError),

    #[error("model not found: {model}. Searched: {searched:?}")]
    ModelNotFound {
        model: String,
        searched: Vec<String>,
    },
}
