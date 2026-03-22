use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};

use crate::coref::CorefResolver;
use crate::ner::{ExtractedEntity, NerEngine, NerError};
use crate::rel::{ExtractedRelation, RelEngine, RelError};
use crate::remap;
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

        // Use the pipeline's confidence_threshold as the GLiNER model-level threshold too.
        // Parameters::default() hardcodes 0.5 which silently drops low-confidence spans
        // before we ever see them; pass our threshold so callers control the cutoff.
        let ner = NerEngine::new(&ner_model, &ner_tokenizer, confidence_threshold as f32)
            .map_err(PipelineError::Ner)?;

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
        // Step 1: NER — extract entities.
        // GLiNER v2.1 span mode works best with schema key names as labels
        // ("Person", "Database", etc.) — these match the model's training vocabulary
        // and produce reliable entity_type values without a description→key mapping.
        // Natural-language descriptions were tested but hurt performance because
        // the model misclassifies service/component names under "programming language"
        // and similar too-generic prompts.
        let labels: Vec<&str> = self.schema.entity_labels();
        let mut entities = self
            .ner
            .extract(text, &labels, None)
            .map_err(PipelineError::Ner)?;

        // Filter by confidence
        entities.retain(|e| e.confidence >= self.confidence_threshold);

        // Step 1b: Coreference resolution — resolve pronouns to preceding entities
        let coref_entities = CorefResolver::resolve(text, &entities);
        entities.extend(coref_entities);

        // Step 1c: Supplement entities — dictionary-based detection for known names
        // that GLiNER missed, boosting recall from ~0.59 toward ~0.75+
        remap::supplement_entities(text, &mut entities);

        // Step 1d: Entity type remapping — fix common GLiNER misclassifications
        // using domain knowledge lookup tables (Database, Infrastructure, Pattern, etc.)
        remap::remap_entity_types(&mut entities);

        // Step 1e: LLM entity cleanup — DISABLED.
        // Tested both local Ollama (qwen2.5:3b) and GPT-4.1-mini for cleaning
        // up GLiNER entity names. Both hurt full benchmark F1:
        //   - Ollama: entity 0.845→0.787, combined 0.678→0.630
        //   - GPT:    entity 0.845→0.811, combined 0.700→0.659
        // Root cause: most entities are already correct; cleanup removes/renames
        // valid multi-word entities like "saga pattern", "Cloudflare Workers".
        // The approach helps ~5 hard cases but hurts ~15 others.
        // Future: try full LLM entity extraction (Graphiti-style) instead of
        // cleanup, or only apply cleanup to entities with low confidence.

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

/// Find the relation extraction model (token-level multitask GLiNER).
///
/// NOTE: gline-rs RelationPipeline requires a **token-level** model (4 inputs:
/// input_ids, attention_mask, words_mask, text_lengths). Span-level models like
/// `gliner_multi-v2.1` are NOT compatible and must not be listed here.
///
/// Compatible model: `knowledgator/gliner-multitask-large-v0.5` (token_level mode).
/// Pre-converted ONNX available from `onnx-community/gliner-multitask-large-v0.5`.
fn find_rel_model(models_dir: &Path) -> Option<PathBuf> {
    let candidates = [
        // INT8 quantized (from onnx-community, downloaded by ModelManager)
        models_dir.join("gliner-multitask-large-v0.5/onnx/model_int8.onnx"),
        // Full precision (from manual conversion via scripts/convert_model.py)
        models_dir.join("gliner-multitask-large-v0.5/onnx/model.onnx"),
        // Legacy flat layout
        models_dir.join("gliner-multitask-large.onnx"),
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
    } else if prefix == "multitask" {
        vec![
            models_dir.join("gliner-multitask-large-v0.5/tokenizer.json"),
            models_dir.join("tokenizer.json"),
        ]
    } else {
        vec![models_dir.join("tokenizer.json")]
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
