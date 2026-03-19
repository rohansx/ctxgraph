use std::path::Path;

use composable::Composable;
use gliner::model::input::relation::schema::RelationSchema;
use gliner::model::input::text::TextInput;
use gliner::model::output::decoded::SpanOutput;
use gliner::model::output::relation::RelationOutput;
use gliner::model::params::Parameters;
use gliner::model::pipeline::relation::RelationPipeline;
use gliner::model::pipeline::token::TokenPipeline;
use orp::model::Model;
use orp::params::RuntimeParameters;
use orp::pipeline::Pipeline;

use crate::ner::ExtractedEntity;
use crate::schema::ExtractionSchema;

/// A relation extracted between two entities.
#[derive(Debug, Clone)]
pub struct ExtractedRelation {
    pub head: String,
    pub relation: String,
    pub tail: String,
    pub confidence: f64,
}

/// Relation extraction engine.
///
/// Supports two modes:
/// - **Model-based**: Uses gline-rs `RelationPipeline` with the multitask ONNX model.
/// - **Heuristic**: Pattern-based extraction when no relation model is available.
pub enum RelEngine {
    ModelBased(ModelBasedRelEngine),
    Heuristic,
}

/// Model-based relation extraction using gline-rs.
///
/// Requires `gliner-multitask-large-v0.5` ONNX model.
pub struct ModelBasedRelEngine {
    model: Model,
    params: Parameters,
    tokenizer_path: String,
}

impl ModelBasedRelEngine {
    pub fn new(model_path: &Path, tokenizer_path: &Path) -> Result<Self, RelError> {
        let runtime_params = RuntimeParameters::default();
        let model = Model::new(
            model_path
                .to_str()
                .ok_or(RelError::InvalidPath(model_path.display().to_string()))?,
            runtime_params,
        )
        .map_err(|e| RelError::ModelLoad(e.to_string()))?;

        Ok(Self {
            model,
            params: Parameters::default(),
            tokenizer_path: tokenizer_path
                .to_str()
                .ok_or(RelError::InvalidPath(
                    tokenizer_path.display().to_string(),
                ))?
                .to_string(),
        })
    }

    pub fn extract(
        &self,
        text: &str,
        labels: &[&str],
        schema: &ExtractionSchema,
    ) -> Result<(Vec<ExtractedEntity>, Vec<ExtractedRelation>), RelError> {
        // Build relation schema from extraction schema
        let mut relation_schema = RelationSchema::new();
        for (rel_name, spec) in &schema.relation_types {
            let heads: Vec<&str> = spec.head.iter().map(|s| s.as_str()).collect();
            let tails: Vec<&str> = spec.tail.iter().map(|s| s.as_str()).collect();
            relation_schema.push_with_allowed_labels(rel_name, &heads, &tails);
        }

        let input = TextInput::from_str(&[text], labels)
            .map_err(|e| RelError::Inference(e.to_string()))?;

        // Step 1: Run NER via TokenPipeline
        let ner_pipeline = TokenPipeline::new(&self.tokenizer_path)
            .map_err(|e| RelError::Inference(e.to_string()))?;
        let ner_composable = ner_pipeline.to_composable(&self.model, &self.params);
        let ner_output: SpanOutput = ner_composable
            .apply(input)
            .map_err(|e| RelError::Inference(e.to_string()))?;

        // Collect entities from NER output
        let mut entities = Vec::new();
        for sequence_spans in &ner_output.spans {
            for span in sequence_spans {
                let span_text = span.text();
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

        // Step 2: Run relation extraction on top of NER output
        let rel_pipeline =
            RelationPipeline::default(&self.tokenizer_path, &relation_schema)
                .map_err(|e| RelError::Inference(e.to_string()))?;
        let rel_composable = rel_pipeline.to_composable(&self.model, &self.params);
        let rel_output: RelationOutput = rel_composable
            .apply(ner_output)
            .map_err(|e| RelError::Inference(e.to_string()))?;

        // Collect relations
        let mut relations = Vec::new();
        for sequence_rels in &rel_output.relations {
            for rel in sequence_rels {
                relations.push(ExtractedRelation {
                    head: rel.subject().to_string(),
                    relation: rel.class().to_string(),
                    tail: rel.object().to_string(),
                    confidence: rel.probability() as f64,
                });
            }
        }

        Ok((entities, relations))
    }
}

impl RelEngine {
    /// Create a model-based engine if the multitask model is available,
    /// otherwise fall back to heuristic mode.
    pub fn new(model_path: Option<&Path>, tokenizer_path: Option<&Path>) -> Result<Self, RelError> {
        match (model_path, tokenizer_path) {
            (Some(mp), Some(tp)) if mp.exists() && tp.exists() => {
                let engine = ModelBasedRelEngine::new(mp, tp)?;
                Ok(Self::ModelBased(engine))
            }
            _ => Ok(Self::Heuristic),
        }
    }

    /// Extract relations between entities.
    pub fn extract(
        &self,
        text: &str,
        entities: &[ExtractedEntity],
        schema: &ExtractionSchema,
    ) -> Result<Vec<ExtractedRelation>, RelError> {
        match self {
            Self::ModelBased(engine) => {
                let labels: Vec<&str> = schema.entity_labels();
                let (_, relations) = engine.extract(text, &labels, schema)?;
                Ok(relations)
            }
            Self::Heuristic => Ok(heuristic_relations(text, entities, schema)),
        }
    }
}

/// Heuristic relation extraction based on text patterns and entity co-occurrence.
fn heuristic_relations(
    text: &str,
    entities: &[ExtractedEntity],
    schema: &ExtractionSchema,
) -> Vec<ExtractedRelation> {
    let lower = text.to_lowercase();
    let mut relations = Vec::new();

    let patterns: &[(&str, &[&str])] = &[
        ("chose", &["chose", "selected", "picked", "went with", "adopted"]),
        ("rejected", &["rejected", "ruled out", "decided against", "dropped"]),
        ("replaced", &["replaced", "migrated from", "switched from", "moved from"]),
        ("depends_on", &["depends on", "relies on", "requires", "built on", "uses"]),
        ("fixed", &["fixed", "resolved", "patched", "repaired", "debugged"]),
        ("introduced", &["introduced", "added", "implemented", "created", "built"]),
        ("deprecated", &["deprecated", "removed", "phased out", "sunset"]),
        ("caused", &["caused", "resulted in", "led to", "triggered"]),
        ("constrained_by", &["constrained by", "limited by", "blocked by", "due to"]),
    ];

    for (relation, keywords) in patterns {
        let rel_spec = match schema.relation_types.get(*relation) {
            Some(spec) => spec,
            None => continue,
        };

        let keyword_found = keywords.iter().any(|kw| lower.contains(kw));
        if !keyword_found {
            continue;
        }

        for head in entities {
            if !rel_spec.head.contains(&head.entity_type) {
                continue;
            }
            for tail in entities {
                if std::ptr::eq(head, tail) {
                    continue;
                }
                if !rel_spec.tail.contains(&tail.entity_type) {
                    continue;
                }
                relations.push(ExtractedRelation {
                    head: head.text.clone(),
                    relation: relation.to_string(),
                    tail: tail.text.clone(),
                    confidence: 0.6,
                });
            }
        }
    }

    relations
}

#[derive(Debug, thiserror::Error)]
pub enum RelError {
    #[error("invalid path: {0}")]
    InvalidPath(String),

    #[error("failed to load model: {0}")]
    ModelLoad(String),

    #[error("inference error: {0}")]
    Inference(String),
}
