//! GLiNER2 — single-model NER + Relation Extraction (Piece W1 in `docs/ROADMAP.md`).
//!
//! **Status: SCAFFOLDED, NOT FUNCTIONAL.**
//!
//! What's blocked (not what we couldn't be bothered to do):
//!
//! - `fastino/gliner2-large-v1` and `fastino/gliner2-base-v1` exist on
//!   HuggingFace (verified 2026-05-14, 530K+ combined downloads, Apache-2.0).
//! - The official weights are **safetensors only**. No ONNX export ships from
//!   Fastino as of this writing.
//! - `gline-rs v1.0.1` (our current GLiNER wrapper crate) does NOT yet support
//!   GLiNER2's architecture — its joint NER+RE head shape is different, and
//!   relations are emitted as ranked typed tuples rather than the span+edge
//!   format `RelationPipeline` returns.
//!
//! What we need to ship GLiNER2 as Tier 1:
//!   1. Convert `fastino/gliner2-large-v1` to ONNX (Python: `optimum-cli export
//!      onnx --model fastino/gliner2-large-v1 --task gliner2 …` — task may
//!      need a custom export config).
//!   2. Write a Rust ORT loader that handles the 5-input GLiNER2 signature
//!      (text_ids, attention_mask, span_idx, label_ids, [optional] task_id).
//!   3. Add output parsing: GLiNER2 emits hierarchical JSON-ish span lists
//!      with explicit entity types AND relation tuples, all keyed on
//!      user-supplied label vocabularies. Map this onto our universal 9/10
//!      taxonomy.
//!   4. Wire into `UniversalPipeline::extract` as the Tier 1 step before
//!      `LlmExtractor::chat_completion`; only call the LLM when the
//!      confidence gate fires.
//!
//! Estimated effort: 1–2 days of focused work, mostly the ONNX export +
//! testing the inference loop matches the Python reference.
//!
//! For now this module is a **placeholder API surface** so the rest of the
//! crate can be wired against the expected shape. Calling
//! `Gliner2Tier::extract` returns `Err(Gliner2Error::NotImplemented)`.

use std::path::Path;

use crate::ner::ExtractedEntity;
use crate::rel::ExtractedRelation;
use crate::schema::ExtractionSchema;

#[derive(Debug, thiserror::Error)]
pub enum Gliner2Error {
    #[error("GLiNER2 not implemented yet — see crate docs for what's blocked")]
    NotImplemented,
    #[error("model file not found at {0}")]
    ModelNotFound(String),
    #[error("inference error: {0}")]
    Inference(String),
}

#[derive(Debug, Clone)]
pub struct Gliner2Result {
    pub entities: Vec<ExtractedEntity>,
    pub relations: Vec<ExtractedRelation>,
    /// Average per-span confidence; the universal pipeline uses this to decide
    /// whether to escalate to the LLM.
    pub avg_confidence: f64,
}

/// Tier-1 single-pass extractor (scaffold).
pub struct Gliner2Tier {
    // Will hold the ORT session + tokenizer once item 3 is implemented.
    _marker: std::marker::PhantomData<()>,
}

impl Gliner2Tier {
    /// Load GLiNER2 from disk. Currently always returns `NotImplemented`;
    /// the placeholder ensures call-sites compile against the expected API
    /// shape so we don't refactor twice.
    pub fn load(_model_path: &Path) -> Result<Self, Gliner2Error> {
        Err(Gliner2Error::NotImplemented)
    }

    /// Run joint NER + RE in a single forward pass over `text`, constrained
    /// to the entity types and relation types declared in `schema`.
    pub fn extract(
        &self,
        _text: &str,
        _schema: &ExtractionSchema,
    ) -> Result<Gliner2Result, Gliner2Error> {
        Err(Gliner2Error::NotImplemented)
    }
}
