use std::path::Path;

use gliner::model::GLiNER;
use gliner::model::input::text::TextInput;
use gliner::model::params::Parameters;
use gliner::model::pipeline::span::SpanMode;
use orp::params::RuntimeParameters;

/// An entity extracted from text by the NER model.
#[derive(Debug, Clone)]
pub struct ExtractedEntity {
    pub text: String,
    pub entity_type: String,
    pub span_start: usize,
    pub span_end: usize,
    pub confidence: f64,
}

/// NER engine wrapping gline-rs GLiNER in span mode.
///
/// Uses `onnx-community/gliner_large-v2.1` (or any span-based GLiNER ONNX model).
pub struct NerEngine {
    model: GLiNER<SpanMode>,
}

impl NerEngine {
    /// Create a new NER engine from model and tokenizer paths.
    ///
    /// - `model_path`: path to `model.onnx` (or `model_int8.onnx`)
    /// - `tokenizer_path`: path to `tokenizer.json`
    /// - `threshold`: minimum GLiNER span probability (0.0–1.0). `Parameters::default`
    ///   uses 0.5 which is too aggressive for domain-specific labels; use a lower
    ///   value like 0.1–0.3 and let the pipeline apply its own threshold on top.
    pub fn new(model_path: &Path, tokenizer_path: &Path, threshold: f32) -> Result<Self, NerError> {
        let params = Parameters::default().with_threshold(threshold);
        let runtime_params = RuntimeParameters::default();

        let model = GLiNER::<SpanMode>::new(
            params,
            runtime_params,
            tokenizer_path
                .to_str()
                .ok_or(NerError::InvalidPath(tokenizer_path.display().to_string()))?,
            model_path
                .to_str()
                .ok_or(NerError::InvalidPath(model_path.display().to_string()))?,
        )
        .map_err(|e| NerError::ModelLoad(e.to_string()))?;

        Ok(Self { model })
    }

    /// Extract entities from text using the given labels.
    ///
    /// `label_to_type` is an optional mapping from GLiNER label string → canonical
    /// entity type key. Pass `None` to use the label string as-is for `entity_type`.
    /// Pass `Some(pairs)` when using natural-language descriptions as labels so the
    /// returned `entity_type` is the short canonical key (e.g. "Database").
    pub fn extract(
        &self,
        text: &str,
        labels: &[&str],
        label_to_type: Option<&std::collections::HashMap<&str, &str>>,
    ) -> Result<Vec<ExtractedEntity>, NerError> {
        let input =
            TextInput::from_str(&[text], labels).map_err(|e| NerError::Inference(e.to_string()))?;

        let output = self
            .model
            .inference(input)
            .map_err(|e| NerError::Inference(e.to_string()))?;

        let mut entities = Vec::new();

        // output.spans is Vec<Vec<Span>> — outer vec is per-sequence
        for sequence_spans in &output.spans {
            for span in sequence_spans {
                // Use character byte offsets from the span directly — avoids the
                // `text.find()` pitfall that always returns the first occurrence.
                let (start, end) = span.offsets();
                let span_text = span.text().to_string();
                let raw_class = span.class();
                let entity_type = match label_to_type {
                    Some(map) => map.get(raw_class).copied().unwrap_or(raw_class),
                    None => raw_class,
                }
                .to_string();

                entities.push(ExtractedEntity {
                    text: span_text,
                    entity_type,
                    span_start: start,
                    span_end: end,
                    confidence: span.probability() as f64,
                });
            }
        }

        Ok(entities)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum NerError {
    #[error("invalid path: {0}")]
    InvalidPath(String),

    #[error("failed to load model: {0}")]
    ModelLoad(String),

    #[error("inference error: {0}")]
    Inference(String),
}
