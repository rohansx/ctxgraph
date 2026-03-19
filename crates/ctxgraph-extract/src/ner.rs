use std::path::Path;

use gliner::model::input::text::TextInput;
use gliner::model::params::Parameters;
use gliner::model::pipeline::span::SpanMode;
use gliner::model::GLiNER;
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
    pub fn new(model_path: &Path, tokenizer_path: &Path) -> Result<Self, NerError> {
        let params = Parameters::default();
        let runtime_params = RuntimeParameters::default();

        let model = GLiNER::<SpanMode>::new(
            params,
            runtime_params,
            tokenizer_path.to_str().ok_or(NerError::InvalidPath(
                tokenizer_path.display().to_string(),
            ))?,
            model_path
                .to_str()
                .ok_or(NerError::InvalidPath(model_path.display().to_string()))?,
        )
        .map_err(|e| NerError::ModelLoad(e.to_string()))?;

        Ok(Self { model })
    }

    /// Extract entities from text using the given labels.
    ///
    /// Returns a list of extracted entities with spans and confidence scores.
    pub fn extract(
        &self,
        text: &str,
        labels: &[&str],
    ) -> Result<Vec<ExtractedEntity>, NerError> {
        let input = TextInput::from_str(&[text], labels)
            .map_err(|e| NerError::Inference(e.to_string()))?;

        let output = self
            .model
            .inference(input)
            .map_err(|e| NerError::Inference(e.to_string()))?;

        let mut entities = Vec::new();

        // output.spans is Vec<Vec<Span>> — outer vec is per-sequence
        for sequence_spans in &output.spans {
            for span in sequence_spans {
                let span_text = span.text();
                // Find the byte offset in the original text
                if let Some(start) = text.find(span_text) {
                    entities.push(ExtractedEntity {
                        text: span_text.to_string(),
                        entity_type: span.class().to_string(),
                        span_start: start,
                        span_end: start + span_text.len(),
                        confidence: span.probability() as f64,
                    });
                }
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
